# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import random
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 26
    max_generations = 100
    population_size = 50
    tournament_size = 5
    mutation_rate_start = 0.1
    mutation_rate_end = 0.01
    elite_count = 5
    
    def create_initial_population(size):
        """Create initial population with multi-scale grid initialization"""
        population = []
        for _ in range(size):
            circles = np.zeros((n_circles, 3))
            
            # Grid-based initialization with adaptive spacing
            grid_size = int(np.ceil(np.sqrt(n_circles)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= n_circles:
                        break
                    x = (i + 1) * spacing_x
                    y = (j + 1) * spacing_y
                    
                    # Add small random perturbation
                    x += np.random.uniform(-spacing_x/4, spacing_x/4)
                    y += np.random.uniform(-spacing_y/4, spacing_y/4)
                    
                    # Ensure within bounds
                    x = max(0.01, min(0.99, x))
                    y = max(0.01, min(0.99, y))
                    
                    # Initial radius: small value
                    r = 0.01
                    circles[idx] = [x, y, r]
                    idx += 1
                if idx >= n_circles:
                    break
            
            # Fill remaining circles with random positions and small radii
            for i in range(idx, n_circles):
                x = np.random.uniform(0.01, 0.99)
                y = np.random.uniform(0.01, 0.99)
                r = 0.01
                circles[i] = [x, y, r]
            
            population.append(circles)
        
        return population
    
    def validate_circles(circles):
        """Validate that circles are within bounds and non-overlapping"""
        # Check containment
        for i in range(n_circles):
            x, y, r = circles[i]
            if r > x or r > y or r > 1-x or r > 1-y:
                return False
        
        # Check overlaps using KDTree for efficiency
        points = circles[:, :2]
        tree = KDTree(points)
        
        for i in range(n_circles):
            x1, y1, r1 = circles[i]
            # Find nearby circles (within 2*(r1+r2) range)
            nearby = tree.query_ball_point([x1, y1], 2*(r1+0.01))
            for j in nearby:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if distance < (r1 + r2):
                        return False
        return True
    
    def calculate_fitness(circles):
        """Calculate fitness as sum of radii"""
        return np.sum(circles[:, 2])
    
    def mutate_circles(circles, generation, max_generations):
        """Apply mutation to circles"""
        # Calculate adaptive mutation rate (exponential decay)
        mutation_rate = mutation_rate_start * (mutation_rate_end/mutation_rate_start) ** (generation/max_generations)
        
        mutated = circles.copy()
        
        # Mutate each circle with some probability
        for i in range(n_circles):
            if np.random.random() < mutation_rate:
                # Randomly choose what to modify
                if np.random.random() < 0.5:
                    # Mutate position
                    dx = np.random.normal(0, 0.02)
                    dy = np.random.normal(0, 0.02)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + dx))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + dy))
                else:
                    # Mutate radius
                    dr = np.random.normal(0, 0.01)
                    mutated[i, 2] = max(0.001, mutated[i, 2] + dr)
        
        return mutated
    
    def repair_circles(circles):
        """Repair invalid circles by adjusting positions/radii"""
        repaired = circles.copy()
        
        # Ensure containment and adjust radii if needed
        for i in range(n_circles):
            x, y, r = repaired[i]
            
            # Adjust radius to fit within bounds
            max_r = min(x, y, 1-x, 1-y)
            if r > max_r:
                r = max_r * 0.99  # Slightly reduce to ensure validity
                repaired[i, 2] = r
            
            # Adjust position if necessary
            if r > x:
                x = r + 0.001
            if r > y:
                y = r + 0.001
            if r > 1-x:
                x = 1 - r - 0.001
            if r > 1-y:
                y = 1 - r - 0.001
                
            repaired[i, 0] = x
            repaired[i, 1] = y
        
        # Handle overlaps
        points = repaired[:, :2]
        tree = KDTree(points)
        
        # Try to resolve overlaps iteratively
        for _ in range(10):  # Limited iterations to avoid infinite loop
            any_changed = False
            for i in range(n_circles):
                x1, y1, r1 = repaired[i]
                
                # Find neighbors
                nearby = tree.query_ball_point([x1, y1], 2*(r1+0.01))
                for j in nearby:
                    if i != j:
                        x2, y2, r2 = repaired[j]
                        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        
                        if distance < (r1 + r2):
                            # Shrink one of them
                            if r1 > r2:
                                new_r1 = max(0.001, r1 - 0.001)
                                repaired[i, 2] = new_r1
                            else:
                                new_r2 = max(0.001, r2 - 0.001)
                                repaired[j, 2] = new_r2
                            any_changed = True
                            
            if not any_changed:
                break
                
        return repaired
    
    def crossover(parent1, parent2):
        """Perform crossover between two parents"""
        child = np.zeros_like(parent1)
        
        # Uniform crossover
        for i in range(n_circles):
            if np.random.random() < 0.5:
                child[i] = parent1[i].copy()
            else:
                child[i] = parent2[i].copy()
        
        return child
    
    def select_tournament(population, fitnesses, tournament_size):
        """Tournament selection"""
        selected_idx = np.random.choice(len(population), tournament_size)
        best_idx = selected_idx[np.argmax([fitnesses[i] for i in selected_idx])]
        return population[best_idx]
    
    # Initialize population
    population = create_initial_population(population_size)
    
    # Evaluate initial population
    fitnesses = []
    for i, circles in enumerate(population):
        # Validate and repair if needed
        if not validate_circles(circles):
            population[i] = repair_circles(circles)
        fitnesses.append(calculate_fitness(population[i]))
    
    # Evolution loop
    for generation in range(max_generations):
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]
        
        # Elitism: keep top individuals
        elites = population[:elite_count]
        
        # Create new population
        new_population = elites[:]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1 = select_tournament(population, fitnesses, tournament_size)
            parent2 = select_tournament(population, fitnesses, tournament_size)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate_circles(child, generation, max_generations)
            
            # Repair if needed
            child = repair_circles(child)
            
            new_population.append(child)
        
        # Update population
        population = new_population[:population_size]
        
        # Recalculate fitnesses
        fitnesses = [calculate_fitness(circles) for circles in population]
        
        # Print progress
        if generation % 20 == 0:
            best_fitness = max(fitnesses)
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

    # Select final best solution
    sorted_indices = np.argsort(fitnesses)[::-1]
    best_solution = population[sorted_indices[0]]
    
    # Final validation and repair
    if not validate_circles(best_solution):
        best_solution = repair_circles(best_solution)
    
    # Apply local optimization refinement
    # Try to slightly increase radii while maintaining constraints
    refined = best_solution.copy()
    
    # Simple local optimization: try to increase radii without overlap
    for _ in range(50):
        improved = False
        for i in range(n_circles):
            # Try small radius increase
            old_r = refined[i, 2]
            new_r = min(old_r + 0.001, min(
                refined[i, 0], refined[i, 1], 1-refined[i, 0], 1-refined[i, 1]
            ))
            
            # Check if we can make this change
            test_circles = refined.copy()
            test_circles[i, 2] = new_r
            
            # Validate
            if validate_circles(test_circles):
                refined = test_circles
                improved = True
                break
        
        if not improved:
            break
    
    return refined

# EVOLVE-BLOCK-END
