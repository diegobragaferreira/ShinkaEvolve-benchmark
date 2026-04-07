# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math

class CirclePackingOptimizer:
    def __init__(self, n_circles=21, rect_width=1.0, rect_height=1.0):
        self.n_circles = n_circles
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.grid_size = 20  # Spatial grid for fast neighbor lookup
        
    def initialize_population(self, population_size=20):
        """Initialize population with diverse starting configurations"""
        population = []
        for _ in range(population_size):
            circles = self._create_hexagonal_initialization()
            # Add some randomness to avoid perfect symmetry
            for i in range(self.n_circles):
                circles[i, 0] += np.random.normal(0, 0.02)
                circles[i, 1] += np.random.normal(0, 0.02)
                circles[i, 2] = max(0.001, circles[i, 2] + np.random.normal(0, 0.01))
            population.append(circles)
        return population
    
    def _create_hexagonal_initialization(self):
        """Create initial hexagonal lattice configuration"""
        circles = np.zeros((self.n_circles, 3))
        
        # Hexagonal packing arrangement
        rows = 4
        cols = 6
        
        spacing_x = self.rect_width / (cols + 1)
        spacing_y = self.rect_height / (rows + 1)
        
        idx = 0
        for i in range(rows):
            offset = spacing_x * (i % 2) * 0.5
            for j in range(cols):
                if idx >= self.n_circles:
                    break
                x = (j + 1) * spacing_x + offset
                y = (i + 1) * spacing_y
                
                # Ensure position is within bounds
                x = max(0.01, min(self.rect_width - 0.01, x))
                y = max(0.01, min(self.rect_height - 0.01, y))
                
                # Initial radius
                circles[idx] = [x, y, 0.05]
                idx += 1
                
            if idx >= self.n_circles:
                break
        
        # Fill remaining circles if needed
        while idx < self.n_circles:
            x = np.random.uniform(0.01, self.rect_width - 0.01)
            y = np.random.uniform(0.01, self.rect_height - 0.01)
            circles[idx] = [x, y, 0.05]
            idx += 1
            
        return circles
    
    def is_valid_solution(self, circles):
        """Check if solution satisfies all constraints"""
        # Check boundary constraints
        for i in range(self.n_circles):
            x, y, r = circles[i]
            if (x - r < 0 or x + r > self.rect_width or 
                y - r < 0 or y + r > self.rect_height):
                return False
        
        # Check collision constraints using spatial grid for efficiency
        grid = self._build_spatial_grid(circles)
        for i in range(self.n_circles):
            neighbors = self._get_neighbors(grid, i, circles)
            for j in neighbors:
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance < circles[i, 2] + circles[j, 2]:
                        return False
        return True
    
    def _build_spatial_grid(self, circles):
        """Build spatial grid for fast neighbor lookup"""
        grid = {}
        cell_size = min(self.rect_width, self.rect_height) / self.grid_size
        
        for i in range(self.n_circles):
            x, y = circles[i, 0], circles[i, 1]
            grid_x = int(x / cell_size)
            grid_y = int(y / cell_size)
            
            if (grid_x, grid_y) not in grid:
                grid[(grid_x, grid_y)] = []
            grid[(grid_x, grid_y)].append(i)
            
        return grid
    
    def _get_neighbors(self, grid, index, circles):
        """Get neighbors using spatial grid"""
        cell_size = min(self.rect_width, self.rect_height) / self.grid_size
        x, y = circles[index, 0], circles[index, 1]
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)
        
        neighbors = []
        
        # Check neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = grid_x + dx, grid_y + dy
                if (nx, ny) in grid:
                    neighbors.extend(grid[(nx, ny)])
                    
        return neighbors
    
    def evaluate_fitness(self, circles):
        """Evaluate fitness (sum of radii) with penalties for constraint violations"""
        if not self.is_valid_solution(circles):
            return -1e10  # Very bad fitness for invalid solutions
        
        return np.sum(circles[:, 2])
    
    def mutate_individual(self, circles, mutation_rate=0.1):
        """Mutate individual with adaptive mutation rates"""
        mutated = circles.copy()
        for i in range(self.n_circles):
            if np.random.random() < mutation_rate:
                # Adaptive mutation based on local density
                neighbors = self._get_neighbors(self._build_spatial_grid(mutated), i, mutated)
                density_factor = len(neighbors) / self.n_circles
                
                # Smaller mutations in dense areas, larger in sparse areas
                mutation_strength = 0.01 * (1.0 - density_factor) + 0.001
                
                # Mutate position
                mutated[i, 0] += np.random.normal(0, mutation_strength)
                mutated[i, 1] += np.random.normal(0, mutation_strength)
                
                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], 0.01, self.rect_width - 0.01)
                mutated[i, 1] = np.clip(mutated[i, 1], 0.01, self.rect_height - 0.01)
                
                # Mutate radius with smaller adjustment
                radius_delta = np.random.normal(0, mutation_strength * 0.5)
                mutated[i, 2] = max(0.001, mutated[i, 2] + radius_delta)
        
        return mutated
    
    def crossover(self, parent1, parent2):
        """Single-point crossover between parents"""
        child = parent1.copy()
        # Crossover points
        crossover_point = np.random.randint(1, self.n_circles)
        
        # Swap circles after crossover point
        child[crossover_point:, :] = parent2[crossover_point:, :]
        
        return child
    
    def optimize(self, max_generations=100, population_size=30):
        """Main optimization loop using evolutionary algorithm"""
        # Initialize population
        population = self.initialize_population(population_size)
        best_fitness = float('-inf')
        best_individual = None
        
        # Evolutionary algorithm
        for generation in range(max_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Print progress
            if generation % 20 == 0:
                print(f"Generation {generation}, Best fitness: {best_fitness}")
            
            # Check for convergence (early stopping)
            if generation > 10:
                recent_improvements = [fitness_scores[i] - fitness_scores[i-1] 
                                     for i in range(1, min(11, len(fitness_scores)))]
                avg_improvement = np.mean(recent_improvements) if recent_improvements else 0
                if avg_improvement < 1e-6:
                    break
            
            # Selection: keep top 50%
            sorted_indices = np.argsort(fitness_scores)[::-1]
            selected_indices = sorted_indices[:population_size//2]
            selected_population = [population[i] for i in selected_indices]
            
            # Generate new population through crossover and mutation
            new_population = selected_population.copy()
            
            # Elitism: keep best individual
            new_population.append(best_individual.copy())
            
            # Create offspring
            while len(new_population) < population_size:
                # Tournament selection
                parent1_idx = np.random.choice(selected_indices)
                parent2_idx = np.random.choice(selected_indices)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate_individual(child)
                
                new_population.append(child)
            
            population = new_population[:population_size]
        
        return best_individual

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle with perimeter = 4, so width + height = 2
    # Using a square for simplicity: width = height = 1
    optimizer = CirclePackingOptimizer(n_circles=21, rect_width=1.0, rect_height=1.0)
    
    # Run optimization
    circles = optimizer.optimize(max_generations=150, population_size=30)
    
    # Local refinement to further improve the solution
    best_fitness = optimizer.evaluate_fitness(circles)
    
    for _ in range(50):
        # Try to improve each circle individually
        improved = False
        for i in range(21):
            # Save original
            original = circles[i].copy()
            
            # Attempt to increase radius
            max_radius = min(
                circles[i][0],  # Distance to left edge
                1.0 - circles[i][0],  # Distance to right edge
                circles[i][1],  # Distance to bottom edge
                1.0 - circles[i][1]   # Distance to top edge
            ) - 0.001
            
            # Consider collision constraints with neighbors
            for j in range(21):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    collision_radius = distance - circles[j][2] - 0.001
                    if collision_radius > 0:
                        max_radius = min(max_radius, collision_radius)
            
            if max_radius > circles[i][2] and max_radius > 0.001:
                # Try to increase radius
                new_radius = min(max_radius, circles[i][2] + 0.005)
                if new_radius > circles[i][2]:
                    circles[i][2] = new_radius
                    improved = True
                    
        # If no improvement, stop
        if not improved:
            break
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")