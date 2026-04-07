# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Fixed seed for deterministic results
np.random.seed(42)
random.seed(42)

class CirclePacker32:
    def __init__(self, n_circles: int = 32):
        self.n_circles = n_circles
        self.bounds = [(0, 1), (0, 1), (0, 0.5)]  # x, y, radius bounds
        
    def initialize_population(self, population_size: int = 100) -> np.ndarray:
        """Initialize a population of circle configurations"""
        population = []
        
        for _ in range(population_size):
            circles = np.zeros((self.n_circles, 3))
            
            # Greedy initialization: place circles one by one
            for i in range(self.n_circles):
                best_circle = self._greedy_place_circle(circles[:i])
                if best_circle is not None:
                    circles[i] = best_circle
                else:
                    # Fallback: random placement if greedy fails
                    circles[i] = self._random_circle()
            
            # Add small perturbation for diversity
            circles += np.random.normal(0, 0.001, circles.shape)
            
            # Ensure validity
            circles = self._ensure_validity(circles)
            population.append(circles)
            
        return np.array(population)
    
    def _random_circle(self) -> np.ndarray:
        """Generate a random valid circle"""
        x = np.random.uniform(0.01, 0.99)
        y = np.random.uniform(0.01, 0.99)
        r = min(min(x, 1-x), min(y, 1-y)) * 0.4  # Ensure containment
        return np.array([x, y, r])
    
    def _greedy_place_circle(self, existing_circles: np.ndarray) -> np.ndarray:
        """Place a new circle greedily maximizing radius"""
        if len(existing_circles) == 0:
            return self._random_circle()
        
        # Find best placement using sampling
        best_radius = 0
        best_pos = None
        
        # Sample potential positions and radii
        for _ in range(1000):
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            
            # Calculate max radius at this position
            max_r = min(min(x, 1-x), min(y, 1-y))
            
            if max_r <= 0:
                continue
                
            # Check overlap with existing circles
            valid = True
            for circ in existing_circles:
                dist = np.sqrt((x - circ[0])**2 + (y - circ[1])**2)
                if dist < (max_r + circ[2]):  # Overlap detected
                    valid = False
                    break
                    
            if valid and max_r > best_radius:
                best_radius = max_r
                best_pos = [x, y, max_r]
                
        return best_pos
    
    def _ensure_validity(self, circles: np.ndarray) -> np.ndarray:
        """Ensure all circles are within bounds and have valid radii"""
        result = circles.copy()
        
        for i in range(len(result)):
            x, y, r = result[i]
            
            # Ensure containment
            r = min(r, x, 1-x, y, 1-y)
            
            # Ensure positive radius
            r = max(r, 0.001)
            
            result[i] = [x, y, r]
            
        return result
    
    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate the fitness of a circle configuration"""
        try:
            # Check containment
            for x, y, r in circles:
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    return -1000  # Invalid - heavily penalized
            
            # Check overlaps using spatial indexing for efficiency
            tree = cKDTree(circles[:, :2])
            pairs = tree.query_pairs(r=0.0001, output_type='ndarray')
            
            # If there are any pairs, check overlap properly
            if len(pairs) > 0:
                for i, j in pairs:
                    if i >= j:
                        continue
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    if dist_sq < min_dist_sq:
                        return -1000  # Overlapping - heavily penalized
            
            # Return sum of radii as fitness
            return np.sum(circles[:, 2])
            
        except Exception:
            return -1000  # Handle edge cases
    
    def mutate(self, circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Apply mutation to a circle configuration"""
        mutated = circles.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Mutate x coordinate
                mutated[i][0] += np.random.normal(0, 0.01)
                # Mutate y coordinate
                mutated[i][1] += np.random.normal(0, 0.01)
                # Mutate radius
                mutated[i][2] += np.random.normal(0, 0.005)
                
                # Ensure valid values
                mutated[i][0] = np.clip(mutated[i][0], 0.01, 0.99)
                mutated[i][1] = np.clip(mutated[i][1], 0.01, 0.99)
                mutated[i][2] = max(mutated[i][2], 0.001)
                
        return mutated
    
    def evolve_generation(self, population: np.ndarray, elite_size: int = 10) -> np.ndarray:
        """Evolve one generation of the population"""
        # Evaluate fitness for all individuals
        fitness_scores = [self.evaluate_fitness(individual) for individual in population]
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = population[sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        elite = sorted_population[:elite_size]
        
        # Select parents (tournament selection)
        new_population = [elite[i % len(elite)] for i in range(len(population))]
        
        # Apply crossover and mutation
        for i in range(len(new_population)):
            if i >= elite_size and np.random.random() < 0.8:  # 80% chance of crossover/mutation
                parent = new_population[i]
                new_population[i] = self.mutate(parent)
        
        return np.array(new_population)
    
    def optimize(self) -> np.ndarray:
        """Main optimization loop"""
        # Stage 1: Initialize with good greedy placements
        population = self.initialize_population(50)
        
        # Get best initial solution
        best_fitness = -float('inf')
        best_solution = None
        
        # Stage 2: Evolutionary optimization
        for generation in range(200):  # 200 generations should be enough
            population = self.evolve_generation(population)
            
            # Track best solution
            current_fitnesses = [self.evaluate_fitness(ind) for ind in population]
            max_fitness_idx = np.argmax(current_fitnesses)
            current_best = population[max_fitness_idx]
            current_fitness = current_fitnesses[max_fitness_idx]
            
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = current_best.copy()
            
            # Early stopping if we're getting close to a good solution
            if best_fitness > 2.9:  # Good threshold to stop early
                break
        
        # Stage 3: Local refinement using gradient-based method or additional search
        if best_solution is not None:
            refined_solution = self.refine_solution(best_solution)
            final_fitness = self.evaluate_fitness(refined_solution)
            
            if final_fitness > best_fitness:
                best_solution = refined_solution
        
        return self._ensure_validity(best_solution) if best_solution is not None else np.zeros((32, 3))
    
    def refine_solution(self, circles: np.ndarray, iterations: int = 50) -> np.ndarray:
        """Apply local refinement to improve solution quality"""
        refined = circles.copy()
        
        for _ in range(iterations):
            # Try small random adjustments
            for i in range(len(refined)):
                old_x, old_y, old_r = refined[i]
                
                # Try small adjustments
                dx = np.random.normal(0, 0.001)
                dy = np.random.normal(0, 0.001)
                dr = np.random.normal(0, 0.0005)
                
                new_x = old_x + dx
                new_y = old_y + dy
                new_r = old_r + dr
                
                # Clip to valid ranges
                new_x = np.clip(new_x, 0.01, 0.99)
                new_y = np.clip(new_y, 0.01, 0.99)
                new_r = max(new_r, 0.001)
                
                # Check if new configuration is valid and better
                temp_config = refined.copy()
                temp_config[i] = [new_x, new_y, new_r]
                
                # Ensure all circles fit within bounds
                valid = True
                for j in range(len(temp_config)):
                    x, y, r = temp_config[j]
                    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                        valid = False
                        break
                
                if valid:
                    # Check overlaps with others (simplified)
                    valid = True
                    for j in range(len(temp_config)):
                        if i != j:
                            x1, y1, r1 = temp_config[i]
                            x2, y2, r2 = temp_config[j]
                            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            if dist < (r1 + r2):
                                valid = False
                                break
                    
                    if valid:
                        # Accept the change if it increases total radius
                        old_total = np.sum(circles[:, 2])
                        new_total = np.sum(temp_config[:, 2])
                        
                        if new_total > old_total:
                            refined = temp_config
        
        return refined

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    packer = CirclePacker32(n_circles=32)
    return packer.optimize()

# EVOLVE-BLOCK-END
