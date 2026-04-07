# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import time
import math

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Optimal rectangle dimensions determined through analysis
RECT_WIDTH = 1.33
RECT_HEIGHT = 0.67

class AdaptiveEvolutionaryCirclePacker:
    def __init__(self, width: float = RECT_WIDTH, height: float = RECT_HEIGHT, num_circles: int = 21):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.rect_area = width * height
        
    def initialize_adaptive_grid(self) -> np.ndarray:
        """Initialize circles using an adaptive grid approach that considers aspect ratio and circle count"""
        circles = np.zeros((self.num_circles, 3))
        
        # Determine grid dimensions based on aspect ratio and circle count
        aspect_ratio = self.width / self.height
        sqrt_n = math.sqrt(self.num_circles)
        
        # Calculate grid dimensions using optimized formulas
        cols = max(1, int(math.ceil(sqrt_n * math.sqrt(aspect_ratio) * 1.15)))
        rows = max(1, int(math.ceil(self.num_circles / cols)))
        
        # Ensure we have enough cells
        if cols * rows < self.num_circles:
            cols = max(cols, int(math.ceil(self.num_circles / rows)))
        
        cell_width = self.width / cols
        cell_height = self.height / rows
        
        # Create grid with proper spacing and controlled randomness
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= self.num_circles:
                    break
                    
                # Position with minimal perturbation for better stability
                x = (j + 0.5) * cell_width + random.uniform(-cell_width*0.08, cell_width*0.08)
                y = (i + 0.5) * cell_height + random.uniform(-cell_height*0.08, cell_height*0.08)
                
                # Keep within bounds with padding
                x = max(0.01, min(self.width - 0.01, x))
                y = max(0.01, min(self.height - 0.01, y))
                
                # Compute maximum possible radius for this position
                max_radius = min(x, self.width - x, y, self.height - y)
                # Use more aggressive initial radius to promote faster convergence
                initial_radius = max(0.01, min(max_radius * 0.38, 0.20))
                
                circles[idx] = [x, y, initial_radius]
                idx += 1
        
        return circles
    
    def is_valid_position(self, x: float, y: float, r: float) -> bool:
        """Check if circle position is within bounds"""
        return (r <= x <= self.width - r and r <= y <= self.height - r)
    
    def calculate_overlap_penalty(self, circles: np.ndarray) -> float:
        """Calculate penalty for overlaps using spatial indexing for efficiency"""
        penalty = 0.0
        
        # Use spatial indexing for efficient neighbor queries
        try:
            points = circles[:, :2]
            tree = cKDTree(points)
            
            # Find pairs that might overlap with more precise bounds
            max_radius = np.max(circles[:, 2]) if len(circles) > 0 else 0.0
            # Tighter bounds for performance
            pairs = tree.query_pairs(2.2 * max_radius, output_type='ndarray')
            
            for i, j in pairs:
                if i < j:  # Avoid double counting
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    
                    # Calculate actual distance
                    dx = x1 - x2
                    dy = y1 - y2
                    distance_sq = dx*dx + dy*dy
                    radius_sum = r1 + r2
                    
                    # If circles overlap
                    if distance_sq < radius_sum * radius_sum:
                        overlap = radius_sum - math.sqrt(distance_sq)
                        # Increased penalty scaling for severe overlaps
                        penalty += overlap * 2000.0
                        
        except Exception:
            # Fallback to brute force if spatial indexing fails
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    
                    dx = x1 - x2
                    dy = y1 - y2
                    distance_sq = dx*dx + dy*dy
                    radius_sum = r1 + r2
                    
                    if distance_sq < radius_sum * radius_sum:
                        overlap = radius_sum - math.sqrt(distance_sq)
                        penalty += overlap * 2000.0
                        
        return penalty
    
    def calculate_boundary_penalty(self, circles: np.ndarray) -> float:
        """Calculate penalty for boundary violations"""
        penalty = 0.0
        
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Calculate how much we're violating the boundary
            if x - r < 0:
                penalty += (0 - (x - r)) * 10000.0
            if x + r > self.width:
                penalty += ((x + r) - self.width) * 10000.0
            if y - r < 0:
                penalty += (0 - (y - r)) * 10000.0
            if y + r > self.height:
                penalty += ((y + r) - self.height) * 10000.0
                
        return penalty
    
    def calculate_fitness(self, circles: np.ndarray) -> Tuple[float, float]:
        """Calculate fitness function including both radius sum and penalties"""
        radius_sum = np.sum(circles[:, 2])
        
        # Calculate penalties
        overlap_penalty = self.calculate_overlap_penalty(circles)
        boundary_penalty = self.calculate_boundary_penalty(circles)
        
        # Combined fitness (higher is better)
        fitness = radius_sum - overlap_penalty - boundary_penalty
        
        return fitness, overlap_penalty + boundary_penalty
    
    def local_optimize_single_circle(self, circles: np.ndarray, idx: int, 
                                   max_attempts: int = 25) -> bool:
        """Optimize a single circle's position and radius using gradient-like approach"""
        old_x, old_y, old_r = circles[idx]
        best_x, best_y, best_r = old_x, old_y, old_r
        best_fitness = float('-inf')
        
        # Get current fitness for comparison
        current_fitness, _ = self.calculate_fitness(circles)
        
        # Try different modifications to this circle
        for attempt in range(max_attempts):
            # Create candidate modifications with strategic patterns
            modifications = []
            
            # Try increasing radius with controlled growth
            if old_r < 0.22:  # Cap maximum radius
                new_r = min(0.22, old_r * (1.0 + random.uniform(0.015, 0.06)))
                modifications.append(('radius', new_r, old_x, old_y))
            
            # Try moving position in various directions with strategic spacing
            directions = [(0, 0), (0.015, 0), (-0.015, 0), (0, 0.015), (0, -0.015),
                         (0.008, 0.008), (0.008, -0.008), (-0.008, 0.008), (-0.008, -0.008)]
            
            for dx, dy in directions:
                new_x = old_x + dx * random.uniform(0.9, 1.1)
                new_y = old_y + dy * random.uniform(0.9, 1.1)
                new_x = max(old_r, min(self.width - old_r, new_x))
                new_y = max(old_r, min(self.height - old_r, new_y))
                modifications.append(('position', old_r, new_x, new_y))
            
            # Try combinations with higher probability
            if random.random() < 0.3:  # 30% chance for combined changes
                new_r = min(0.22, old_r * (1.0 + random.uniform(-0.02, 0.05)))
                dx = random.uniform(-0.015, 0.015)
                dy = random.uniform(-0.015, 0.015)
                new_x = old_x + dx
                new_y = old_y + dy
                new_x = max(new_r, min(self.width - new_r, new_x))
                new_y = max(new_r, min(self.height - new_r, new_y))
                modifications.append(('combined', new_r, new_x, new_y))
            
            # Test all modifications
            for mod_type, new_r, new_x, new_y in modifications:
                # Temporarily modify this circle
                circles[idx] = [new_x, new_y, new_r]
                
                # Check if modification is valid (within bounds)
                if self.is_valid_position(new_x, new_y, new_r):
                    # Calculate fitness with this modification
                    modified_fitness, _ = self.calculate_fitness(circles)
                    
                    # Accept improvement with better probability or occasional exploration
                    if modified_fitness > best_fitness or (
                        random.random() < 0.02 and modified_fitness > current_fitness):
                        best_fitness = modified_fitness
                        best_x, best_y, best_r = new_x, new_y, new_r
                
                # Restore original values
                circles[idx] = [old_x, old_y, old_r]
        
        # Apply best modification if it improved fitness
        if best_fitness > current_fitness:
            circles[idx] = [best_x, best_y, best_r]
            return True
        
        return False
    
    def local_optimize_all(self, circles: np.ndarray, max_iterations: int = 120) -> bool:
        """Perform local optimization on all circles"""
        improved = False
        
        for iteration in range(max_iterations):
            # Shuffle indices for random order optimization
            indices = list(range(len(circles)))
            random.shuffle(indices)
            
            iteration_improved = False
            for idx in indices:
                if self.local_optimize_single_circle(circles, idx):
                    iteration_improved = True
                    improved = True
            
            # Stop early if no improvement
            if not iteration_improved:
                break
                
        return improved
    
    def optimize(self, max_generations: int = 700) -> np.ndarray:
        """Main optimization routine combining initialization, local optimization, and evolutionary refinement"""
        # Phase 1: Adaptive grid initialization
        circles = self.initialize_adaptive_grid()
        
        # Phase 2: Intensive local optimization
        self.local_optimize_all(circles, max_iterations=120)
        
        # Phase 3: Evolutionary refinement with hybrid approach
        best_circles = circles.copy()
        best_fitness, _ = self.calculate_fitness(best_circles)
        
        # Evolutionary parameters
        population_size = 60
        elite_count = 7
        initial_mutation_rate = 0.15
        
        # Create initial population with more diversity
        population = [best_circles.copy()]
        for _ in range(population_size - 1):
            # Create diverse variants with balanced perturbations
            variant = best_circles.copy()
            for i in range(len(variant)):
                if random.random() < 0.2:  # 20% mutation rate (lower than previous)
                    # Apply small random perturbations
                    variant[i][0] += random.uniform(-0.02, 0.02)
                    variant[i][1] += random.uniform(-0.02, 0.02)
                    variant[i][2] *= random.uniform(0.95, 1.05)
                    
                    # Keep within bounds
                    variant[i][0] = max(variant[i][2], min(self.width - variant[i][2], variant[i][0]))
                    variant[i][1] = max(variant[i][2], min(self.height - variant[i][2], variant[i][1]))
                    variant[i][2] = max(0.001, min(0.22, variant[i][2]))
            
            population.append(variant)
        
        # Evolutionary loop with more intelligent parameter adaptation
        for generation in range(max_generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness, _ = self.calculate_fitness(individual)
                fitness_scores.append(fitness)
            
            # Update best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_circles = population[max_fitness_idx].copy()
            
            # Selection with tournament
            selected = []
            for _ in range(population_size):
                tournament_size = 3
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx].copy())
            
            # Elitism - keep best individuals
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elite = [population[i].copy() for i in elite_indices]
            
            # Create new population - start with elites
            new_population = elite.copy() 
            
            # Fill remaining slots with offspring
            while len(new_population) < population_size:
                # Select parents
                parent1 = selected[random.randint(0, len(selected)-1)]
                parent2 = selected[random.randint(0, len(selected)-1)]
                
                # Create child through crossover
                child = parent1.copy()
                
                # Uniform crossover with more balanced bias
                parent1_fitness = np.sum(parent1[:, 2])
                parent2_fitness = np.sum(parent2[:, 2])
                bias = 0.7 if parent1_fitness > parent2_fitness else 0.3
                
                # Apply crossover
                for i in range(len(parent1)):
                    if random.random() < bias:
                        child[i] = parent1[i].copy()
                    else:
                        child[i] = parent2[i].copy()
                
                # Apply mutation with adaptive rate
                adaptive_mutation_rate = initial_mutation_rate * (1.0 - generation/max_generations * 0.7)
                for i in range(len(child)):
                    if random.random() < adaptive_mutation_rate:
                        # Mutate position or radius
                        if random.random() < 0.7:
                            # Mutate position with controlled magnitude
                            child[i][0] += random.uniform(-0.012, 0.012)
                            child[i][1] += random.uniform(-0.012, 0.012)
                            
                            # Keep within bounds
                            child[i][0] = max(child[i][2], min(self.width - child[i][2], child[i][0]))
                            child[i][1] = max(child[i][2], min(self.height - child[i][2], child[i][1]))
                        else:
                            # Mutate radius with smaller adjustments
                            child[i][2] *= random.uniform(0.93, 1.07)
                            child[i][2] = max(0.001, min(0.22, child[i][2]))
                
                new_population.append(child)
            
            population = new_population[:population_size]
        
        # Final local optimization
        self.local_optimize_all(best_circles, max_iterations=60)
        
        return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    packer = AdaptiveEvolutionaryCirclePacker(width=RECT_WIDTH, height=RECT_HEIGHT, num_circles=21)
    circles = packer.optimize()
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")