# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import math
from typing import Tuple, List
import deap.base
from deap import creator, tools, algorithms
import time
from functools import partial

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility - important for consistent benchmarking
    random.seed(42)
    np.random.seed(42)
    
    # Rectangle dimensions: width + height = 2, using 1.5 x 0.5 for good aspect ratio
    rect_width, rect_height = 1.5, 0.5
    
    # Helper functions
    def create_strategic_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using strategic corner and edge placements"""
        circles = np.zeros((n_circles, 3))
        
        # Strategic corner and edge positions
        strategic_positions = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
            (width * 0.25, height * 0.25),   # quarter positions
            (width * 0.75, height * 0.25),
            (width * 0.25, height * 0.75),
            (width * 0.75, height * 0.75),
        ]
        
        # Place circles at strategic positions first
        for i in range(min(len(strategic_positions), n_circles)):
            x, y = strategic_positions[i]
            circles[i] = [x, y, 0.03]
        
        # Fill remaining positions with hexagonal grid pattern
        remaining = n_circles - len(strategic_positions)
        if remaining > 0:
            # Use hexagonal grid for remaining positions
            rows = int(math.ceil(math.sqrt(remaining)))
            cols = int(math.ceil(remaining / rows))
            
            cell_width = width / (cols + 1)
            cell_height = height / (rows + 1)
            
            idx = len(strategic_positions)
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x_offset = 0.0 if i % 2 == 0 else 0.5
                    x = (j + 1 + x_offset) * cell_width
                    y = (i + 1) * cell_height
                    # Ensure within bounds
                    x = max(0.01, min(width - 0.01, x))
                    y = max(0.01, min(height - 0.01, y))
                    circles[idx] = [x, y, 0.02]
                    idx += 1
                    if idx >= n_circles:
                        break
        
        return circles
    
    def compute_max_radius_at_position(x: float, y: float, existing_circles: np.ndarray, 
                                     rect_width: float, rect_height: float) -> float:
        """Compute maximum possible radius for a circle at given position"""
        # Distance to boundaries
        min_bound = min(x, rect_width - x, y, rect_height - y)
        
        # Distance to other circles - optimized version
        min_dist = float('inf')
        if len(existing_circles) > 0:
            # Vectorized distance calculation using scipy for performance
            positions = existing_circles[:, :2]
            dx = positions[:, 0] - x
            dy = positions[:, 1] - y
            distances = np.sqrt(dx*dx + dy*dy)
            # Avoid self-distance by setting large value
            distances = np.where(distances == 0, float('inf'), distances)
            if len(distances) > 0:
                min_dist = np.min(distances)
                # Subtract radii of nearby circles
                radii = existing_circles[:, 2]
                min_dist = min(min_dist, np.min(distances - radii[distances > 0]))
        
        # Take minimum of boundary and other-circle distances
        max_radius = min(min_bound, min_dist if min_dist < float('inf') else float('inf'))
        return max(0.001, max_radius)
    
    def calculate_radius_sum(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])
    
    def is_valid_configuration(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Check if configuration is valid (no overlaps, all within bounds)"""
        # Check boundary constraints efficiently
        if np.any(circles[:, 0] - circles[:, 2] < 0) or \
           np.any(circles[:, 0] + circles[:, 2] > rect_width) or \
           np.any(circles[:, 1] - circles[:, 2] < 0) or \
           np.any(circles[:, 1] + circles[:, 2] > rect_height):
            return False
        
        # Check overlap constraints efficiently
        if len(circles) < 2:
            return True
            
        # Use vectorized computation for overlap detection
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Create distance matrix
        dist_matrix = cdist(positions, positions)
        
        # Create mask for pairs that could overlap
        # Diagonal should be zero (self-distances), so set to infinity
        np.fill_diagonal(dist_matrix, float('inf'))
        
        # Check if any pair violates overlap constraint
        min_distances = np.min(dist_matrix, axis=1)
        min_radii = np.min(radii[:, np.newaxis] + radii[np.newaxis, :], axis=0)
        
        # Compare minimum distances with sum of radii
        overlap_mask = min_distances < min_radii
        
        return not np.any(overlap_mask)
    
    def evaluate_fitness(individual: List[float]) -> Tuple[float,]:
        """Evaluate fitness of individual (negative sum of radii for minimization)"""
        # Convert flat array back to circles format
        circles = np.array(individual).reshape(-1, 3)
        
        # Check validity
        if not is_valid_configuration(circles, rect_width, rect_height):
            # Large penalty for invalid configurations
            return (-1000000.0,)
        
        # Return negative sum of radii (minimization problem)
        return (-calculate_radius_sum(circles),)
    
    def mutator(individual: List[float], indpb: float = 0.1) -> Tuple[List[float],]:
        """Mutator function for evolutionary algorithm"""
        for i in range(len(individual)):
            if random.random() < indpb:
                # For positions (x,y) - larger mutation
                if i % 3 < 2:
                    individual[i] += random.gauss(0, 0.05)
                    # Keep within bounds
                    individual[i] = max(0.01, min(rect_width - 0.01, individual[i]))
                else:
                    # For radius (r) - smaller mutation
                    individual[i] += random.gauss(0, 0.02)
                    # Keep positive
                    individual[i] = max(0.001, individual[i])
        return (individual,)
    
    def crossover(ind1: List[float], ind2: List[float]) -> Tuple[List[float], List[float]]:
        """Crossover function for evolutionary algorithm"""
        size = len(ind1)
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        return ind1, ind2
    
    # Create individual and population types
    creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Initialize population with multiple strategies
    def initialize_population(size: int) -> List[creator.Individual]:
        population = []
        for _ in range(size):
            # Use hybrid initialization
            circles = create_strategic_initialization(21, rect_width, rect_height)
            
            # Add some randomness
            for i in range(len(circles)):
                if random.random() > 0.7:
                    circles[i, 0] += random.uniform(-0.1, 0.1)
                    circles[i, 1] += random.uniform(-0.1, 0.1)
                    circles[i, 0] = max(0.01, min(rect_width - 0.01, circles[i, 0]))
                    circles[i, 1] = max(0.01, min(rect_height - 0.01, circles[i, 1]))
            
            # Flatten and create individual
            individual = creator.Individual(circles.flatten().tolist())
            population.append(individual)
        return population
    
    # Main evolutionary optimization
    toolbox = deap.base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     lambda: random.uniform(0.01, rect_width - 0.01), 63)
    toolbox.register("population", initialize_population, size=20)
    
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutator)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolutionary optimization
    population = toolbox.population()
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Evolution parameters
    n_generations = 50
    pop_size = 20
    cxpb = 0.7
    mutpb = 0.3
    
    # Run evolution
    hof = tools.HallOfFame(1)
    
    try:
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=cxpb, mutpb=mutpb, 
            ngen=n_generations, stats=stats, halloffame=hof, verbose=False
        )
        
        # Extract best individual
        best_individual = hof[0]
        best_circles = np.array(best_individual).reshape(-1, 3)
        
    except Exception as e:
        # Fallback to hybrid initialization if evolution fails
        best_circles = create_strategic_initialization(21, rect_width, rect_height)
        # Apply some improvement steps
        for _ in range(100):
            for i in range(21):
                # Simple local search around each circle
                current_x, current_y, current_r = best_circles[i]
                
                # Try small perturbations
                new_x = max(0.01, min(rect_width - 0.01, current_x + random.uniform(-0.05, 0.05)))
                new_y = max(0.01, min(rect_height - 0.01, current_y + random.uniform(-0.05, 0.05)))
                
                # Compute maximum radius at new position
                temp_circles = best_circles.copy()
                temp_circles[i] = [new_x, new_y, 0.01]
                max_r = compute_max_radius_at_position(new_x, new_y, temp_circles, rect_width, rect_height)
                new_r = min(max_r, max(0.001, current_r + random.uniform(-0.02, 0.02)))
                
                temp_circles[i] = [new_x, new_y, new_r]
                
                # Validate and accept improvement
                if is_valid_configuration(temp_circles, rect_width, rect_height):
                    new_sum = calculate_radius_sum(temp_circles)
                    old_sum = calculate_radius_sum(best_circles)
                    if new_sum > old_sum:
                        best_circles = temp_circles.copy()
    
    # Final validation and cleanup
    final_circles = best_circles.copy()
    
    # Ensure all circles are valid and apply final local refinement
    for _ in range(50):
        improved = False
        for i in range(21):
            current_x, current_y, current_r = final_circles[i]
            
            # Try small perturbations
            test_x = max(0.01, min(rect_width - 0.01, current_x + random.uniform(-0.02, 0.02)))
            test_y = max(0.01, min(rect_height - 0.01, current_y + random.uniform(-0.02, 0.02)))
            
            # Compute maximum radius
            temp_circles = final_circles.copy()
            temp_circles[i] = [test_x, test_y, 0.01]
            max_r = compute_max_radius_at_position(test_x, test_y, temp_circles, rect_width, rect_height)
            test_r = min(max_r, max(0.001, current_r + random.uniform(-0.01, 0.01)))
            
            temp_circles[i] = [test_x, test_y, test_r]
            
            # Validate and accept improvement
            if is_valid_configuration(temp_circles, rect_width, rect_height):
                new_sum = calculate_radius_sum(temp_circles)
                old_sum = calculate_radius_sum(final_circles)
                if new_sum > old_sum:
                    final_circles = temp_circles.copy()
                    improved = True
        
        # Stop if no improvements in a cycle
        if not improved:
            break
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")