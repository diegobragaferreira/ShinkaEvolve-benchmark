# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import random
import time
from collections import deque
import itertools

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions: width + height = 2, use optimal aspect ratio
    rect_width = 1.3333333333333333  # 2/3
    rect_height = 0.6666666666666666  # 1/3

    n = 21

    # Generate hexagonal grid-based initial placement
    def generate_hexagonal_grid(num_circles, width, height):
        # Create a hexagonal grid that optimally fills the rectangle
        # Calculate grid dimensions based on circle count
        rows = int(np.ceil(np.sqrt(num_circles * 2)))
        cols = int(np.ceil(num_circles / rows))
        
        # Ensure we have enough grid cells
        rows = max(rows, 3)
        cols = max(cols, 3)
        
        # Calculate hexagon parameters
        hex_radius = min(width / cols, height / rows) * 0.45  # Leave margin
        spacing_x = hex_radius * 2
        spacing_y = hex_radius * np.sqrt(3)
        
        # Adjust for rectangle dimensions
        actual_cols = int(width / spacing_x)
        actual_rows = int(height / spacing_y)
        
        circles = []
        y_offset = spacing_y * 0.5  # Start from center of first row
        x_offset = spacing_x * 0.5
        
        circle_count = 0
        for i in range(actual_rows):
            y = y_offset + i * spacing_y
            # Offset every other row
            x_start = x_offset + (i % 2) * spacing_x * 0.5
            for j in range(actual_cols):
                if circle_count >= num_circles:
                    break
                x = x_start + j * spacing_x
                # Check if within bounds
                if x >= hex_radius and x <= width - hex_radius and \
                   y >= hex_radius and y <= height - hex_radius:
                    # Start with reasonable radius
                    r = min(hex_radius * 0.7, 0.1)
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= num_circles:
                break
        
        # Fill remaining slots with random positions
        while len(circles) < num_circles:
            x = np.random.uniform(hex_radius, width - hex_radius)
            y = np.random.uniform(hex_radius, height - hex_radius)
            r = np.random.uniform(0.01, 0.1)
            circles.append([x, y, r])
            
        return np.array(circles)

    # Fitness calculation with penalty for constraints
    def calculate_fitness(circles_array):
        total_radius = np.sum(circles_array[:, 2])
        
        penalty = 0
        
        # Boundary penalties - large penalties for going out of bounds
        for i in range(n):
            cx, cy, r = circles_array[i]
            if cx - r < 0.001:
                penalty += 10000 * (r - cx)**2
            if cx + r > rect_width - 0.001:
                penalty += 10000 * (cx + r - rect_width)**2
            if cy - r < 0.001:
                penalty += 10000 * (r - cy)**2
            if cy + r > rect_height - 0.001:
                penalty += 10000 * (cy + r - rect_height)**2

        # Overlap penalties - check all pairs efficiently using spatial indexing
        points = circles_array[:, :2]
        tree = KDTree(points)
        
        for i in range(n):
            cx, cy, r = circles_array[i]
            # Find neighbors within 2*(r + safety_margin) distance
            neighbor_indices = tree.query_ball_point([cx, cy], 2*r + 0.01)
            
            for j in neighbor_indices:
                if i != j:
                    other_cx, other_cy, other_r = circles_array[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist
                    if overlap > 0:
                        penalty += 100000 * overlap**2
                        
        return total_radius - penalty

    # Enhanced local refinement with better constraint handling
    def refine_circles(circles_array, max_iter=100):
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness(best_circles)
        
        for iteration in range(max_iter):
            improved = False
            # Try to improve each circle
            for i in range(n):
                cx, cy, r = best_circles[i]
                
                # Calculate maximum allowable radius
                max_radius = float('inf')
                
                # Boundary constraints
                max_radius = min(max_radius, cx - 0.001)
                max_radius = min(max_radius, rect_width - cx - 0.001)
                max_radius = min(max_radius, cy - 0.001)
                max_radius = min(max_radius, rect_height - cy - 0.001)
                
                # Overlap constraints with neighbors using spatial indexing
                points = best_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
                
                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)
                
                # Try to increase radius if beneficial
                if max_radius > r and max_radius > 0.001:
                    # Test multiple increment sizes
                    increments = [0.002, 0.005, 0.01, 0.02]
                    for incr in increments:
                        new_r = min(r + incr, max_radius)
                        if new_r <= r:
                            continue
                            
                        # Validate with full constraint check
                        valid = True
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r
                        
                        # Quick neighbor check
                        points_new = temp_circles[:, :2]
                        tree_new = KDTree(points_new)
                        neighbor_indices_new = tree_new.query_ball_point([cx, cy], 2*(new_r + 0.01) + 0.001)
                        
                        for k in neighbor_indices_new:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                                if dist < new_r + other_r:
                                    valid = False
                                    break
                                    
                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 2] = new_r
                            test_fitness = calculate_fitness(test_circles)
                            
                            if test_fitness > best_fitness:
                                best_circles = test_circles
                                best_fitness = test_fitness
                                improved = True
                                break
            
            if not improved:
                break
                
        return best_circles

    # Generate initial solution with hexagonal grid
    initial_circles = generate_hexagonal_grid(n, rect_width, rect_height)
    
    # Initial refinement
    best_circles = refine_circles(initial_circles, max_iter=50)
    best_fitness = calculate_fitness(best_circles)

    # Genetic Algorithm for circle packing optimization
    class CirclePackingGA:
        def __init__(self, population_size=40, generations=50, elite_size=5, mutation_rate=0.15):
            self.population_size = population_size
            self.generations = generations
            self.elite_size = elite_size
            self.mutation_rate = mutation_rate
            self.rect_width = rect_width
            self.rect_height = rect_height
            
        def create_individual(self, base_solution=None):
            """Create a new individual either from base solution or random"""
            if base_solution is not None:
                # Create variant of base solution with mutations
                individual = base_solution.copy()
                # Add noise to each circle
                for i in range(n):
                    # Position noise
                    individual[i, 0] += np.random.normal(0, 0.02)
                    individual[i, 1] += np.random.normal(0, 0.02)
                    # Radius noise
                    individual[i, 2] *= np.random.uniform(0.9, 1.1)
                    
                    # Clamp to bounds
                    individual[i, 0] = np.clip(individual[i, 0], 0.01, self.rect_width - 0.01)
                    individual[i, 1] = np.clip(individual[i, 1], 0.01, self.rect_height - 0.01)
                    individual[i, 2] = max(0.001, individual[i, 2])
            else:
                # Create entirely random individual
                individual = np.zeros((n, 3))
                for i in range(n):
                    x = np.random.uniform(0.01, self.rect_width - 0.01)
                    y = np.random.uniform(0.01, self.rect_height - 0.01)
                    r = np.random.uniform(0.01, 0.1)
                    individual[i] = [x, y, r]
            return individual
            
        def evaluate(self, individual):
            """Evaluate fitness of individual"""
            return calculate_fitness(individual)
            
        def crossover(self, parent1, parent2):
            """Uniform crossover between parents"""
            child = parent1.copy()
            for i in range(n):
                if random.random() > 0.5:
                    child[i] = parent2[i]
            return child
            
        def mutate(self, individual, generation):
            """Mutation operator for circle packing"""
            mutated = individual.copy()
            
            # Adaptive mutation rate
            adaptive_rate = self.mutation_rate * (1.0 - generation / self.generations)
            
            for i in range(n):
                if random.random() < adaptive_rate:
                    # Choose what to mutate
                    mutation_type = random.choice(['position', 'radius'])
                    
                    if mutation_type == 'position':
                        # Mutate position
                        mutated[i, 0] += np.random.normal(0, 0.03)
                        mutated[i, 1] += np.random.normal(0, 0.03)
                        # Clamp to bounds
                        mutated[i, 0] = np.clip(mutated[i, 0], 0.01, self.rect_width - 0.01)
                        mutated[i, 1] = np.clip(mutated[i, 1], 0.01, self.rect_height - 0.01)
                    else:
                        # Mutate radius
                        mutated[i, 2] *= np.random.uniform(0.8, 1.2)
                        mutated[i, 2] = max(0.001, mutated[i, 2])
                        
            return mutated
            
        def run(self):
            """Run the genetic algorithm"""
            # Initialize population
            population = [self.create_individual(best_circles)]
            for _ in range(self.population_size - 1):
                population.append(self.create_individual())
                
            for gen in range(self.generations):
                # Evaluate fitness of population
                fitnesses = [self.evaluate(ind) for ind in population]
                
                # Sort by fitness (descending)
                sorted_indices = np.argsort(fitnesses)[::-1]
                population = [population[i] for i in sorted_indices]
                fitnesses = [fitnesses[i] for i in sorted_indices]
                
                # Update best solution
                if fitnesses[0] > best_fitness:
                    best_fitness = fitnesses[0]
                    best_solution = population[0].copy()
                
                # Create new population with elitism
                new_population = population[:self.elite_size]
                
                # Generate offspring
                while len(new_population) < self.population_size:
                    # Tournament selection
                    parent1 = self.tournament_select(population, fitnesses)
                    parent2 = self.tournament_select(population, fitnesses)
                    
                    # Crossover
                    child = self.crossover(parent1, parent2)
                    
                    # Mutation
                    child = self.mutate(child, gen)
                    
                    new_population.append(child)
                    
                population = new_population[:self.population_size]
                
            return population[0]
            
        def tournament_select(self, population, fitnesses, tournament_size=3):
            """Tournament selection for parent selection"""
            selected_indices = random.sample(range(len(population)), tournament_size)
            selected_fitnesses = [fitnesses[i] for i in selected_indices]
            winner_index = selected_indices[np.argmax(selected_fitnesses)]
            return population[winner_index]

    # Run genetic algorithm optimization
    try:
        ga = CirclePackingGA(population_size=40, generations=40, elite_size=5, mutation_rate=0.15)
        evolved_solution = ga.run()
        
        # Final refinement step
        final_solution = refine_circles(evolved_solution, max_iter=30)
        
        # Final fitness check
        final_fitness = calculate_fitness(final_solution)
        if final_fitness > best_fitness:
            best_solution = final_solution
        else:
            # Do one final refinement of best solution
            best_solution = refine_circles(best_solution, max_iter=20)
            
    except Exception as e:
        # Fallback to best solution found so far
        pass

    # Final safety validation
    final_fitness = calculate_fitness(best_solution)
    if final_fitness < 0:
        # If still invalid, do comprehensive refinement
        best_solution = refine_circles(best_solution, max_iter=50)
        
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")