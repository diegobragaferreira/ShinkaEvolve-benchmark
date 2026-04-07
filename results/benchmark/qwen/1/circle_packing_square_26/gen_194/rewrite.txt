# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools
from numba import jit
import time

# Constants for the algorithm
N_CIRCLES = 26
BENCHMARK = 2.6358627564136983

@jit(nopython=True)
def validate_solution_jit(circles):
    """Fast validation using numba for the critical path"""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        # Check containment
        if r > x or r > y or r > 1-x or r > 1-y:
            return False
        # Check overlap with previous circles
        for j in range(i):
            x2, y2, r2 = circles[j]
            dx = x - x2
            dy = y - y2
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (r+r2)*(r+r2)
            if dist_sq < min_dist_sq:
                return False
    return True

def create_voronoi_initialization(n_circles: int, seed: int = 42) -> np.ndarray:
    """
    Create high-quality initial placement using Voronoi-based method
    that ensures good spatial distribution from the start
    """
    np.random.seed(seed)
    
    # Generate initial points in a structured way
    # Use a combination of grid and boundary sampling to ensure good coverage
    grid_size = int(np.ceil(np.sqrt(n_circles * 1.5)))
    
    # Create grid points with some jitter
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n_circles:
                # Add jitter to improve distribution
                jitter_x = np.random.uniform(-0.03, 0.03)
                jitter_y = np.random.uniform(-0.03, 0.03)
                x = (j + 0.5 + jitter_x) / grid_size
                y = (i + 0.5 + jitter_y) / grid_size
                points.append([x, y])
    
    # Add boundary points for edge coverage
    boundary_points = []
    for _ in range(15):
        side = np.random.randint(0, 4)
        if side == 0:  # Top
            boundary_points.append([np.random.uniform(0.1, 0.9), 1.0])
        elif side == 1:  # Bottom
            boundary_points.append([np.random.uniform(0.1, 0.9), 0.0])
        elif side == 2:  # Left
            boundary_points.append([0.0, np.random.uniform(0.1, 0.9)])
        else:  # Right
            boundary_points.append([1.0, np.random.uniform(0.1, 0.9)])
    
    points.extend(boundary_points)
    
    # Ensure we have enough points
    points = np.array(points[:n_circles])
    
    # If we don't have enough, fill with random points in valid range
    while len(points) < n_circles:
        points = np.vstack([points, [np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)]])
    
    points = points[:n_circles]
    
    # Make sure points are within bounds
    points = np.clip(points, 0.05, 0.95)
    
    return points

def compute_voronoi_based_radii(positions, n_circles: int) -> np.ndarray:
    """
    Compute radii based on Voronoi cell properties to ensure good distribution
    """
    try:
        # Create Voronoi diagram
        vor = Voronoi(positions)
        
        # Compute Voronoi cell areas (approximated)
        radii = np.zeros(n_circles)
        
        # For each point, find its Voronoi cell area and calculate appropriate radius
        for i in range(n_circles):
            if i < len(vor.points):
                x, y = vor.points[i]
                
                # Calculate minimum distance to neighboring points
                min_dist = float('inf')
                for j in range(n_circles):
                    if i != j:
                        dist = np.sqrt((x - positions[j, 0])**2 + (y - positions[j, 1])**2)
                        min_dist = min(min_dist, dist)
                
                # Conservative radius based on Voronoi cell size
                if min_dist > 0:
                    # Use 1/4 of average distance to neighbors as radius
                    radii[i] = min(0.15, min_dist * 0.25)
                else:
                    # Fallback
                    radii[i] = np.random.uniform(0.02, 0.08)
                    
        # Ensure radii respect boundaries
        for i in range(n_circles):
            boundary_safe = min(positions[i, 0], 1-positions[i, 0], 
                               positions[i, 1], 1-positions[i, 1])
            radii[i] = min(radii[i], boundary_safe * 0.9)
            radii[i] = max(0.005, radii[i])
            
    except Exception:
        # Fallback to simple method
        radii = np.array([np.random.uniform(0.02, 0.08) for _ in range(n_circles)])
    
    return radii

def local_radius_optimization(circles):
    """
    Perform efficient local optimization to increase radii while maintaining validity
    Uses geometric constraints instead of expensive binary search
    """
    n = len(circles)
    circles = circles.copy()
    
    # Order circles by their potential to gain radius (largest potential first)
    potential_gains = []
    for i in range(n):
        x, y, r = circles[i]
        # Find minimum distance to neighbors
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                min_dist = min(min_dist, dist)
        
        # Potential gain is how much we could increase radius
        max_gain = min_dist - r if min_dist > r else 0
        potential_gains.append((max_gain, i))
    
    # Sort by potential gain
    potential_gains.sort(reverse=True)
    
    # Try to increase radii for circles with highest potential
    for _, i in potential_gains:
        x, y, r = circles[i]
        max_possible_radius = min(x, y, 1-x, 1-y)
        
        if r >= max_possible_radius:
            continue
            
        # Find minimum distance to neighbors
        min_dist_to_others = float('inf')
        for j in range(n):
            if i != j:
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                min_dist_to_others = min(min_dist_to_others, dist)
        
        # Calculate maximum allowable radius
        if min_dist_to_others > 0:
            max_radius = min(max_possible_radius, min_dist_to_others - 0.001)
            # Increase radius by a safe amount
            new_radius = min(max_radius, r + 0.005)
            if new_radius > r:
                circles[i, 2] = new_radius
                
    return circles

def constrained_position_update(circles, max_iter=3):
    """
    Update positions respecting boundaries and overlaps more efficiently
    """
    n = len(circles)
    circles = circles.copy()
    
    # Simple repulsion-based update
    for iteration in range(max_iter):
        updated = False
        for i in range(n):
            x, y, r = circles[i]
            
            # Apply boundary constraints
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            circles[i, 0] = x
            circles[i, 1] = y
            
            # Try to resolve overlaps with neighbors
            for j in range(n):
                if i != j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    # If overlapping, move them apart
                    if dist < r1 + r2:
                        # Calculate separation vector
                        dx = x2 - x1
                        dy = y2 - y1
                        length = np.sqrt(dx*dx + dy*dy)
                        
                        if length > 0:
                            # Normalize
                            dx /= length
                            dy /= length
                            
                            # Separate by overlap amount
                            overlap = (r1 + r2) - dist
                            move_amount = overlap * 0.5
                            
                            # Apply movement (distribute between both circles)
                            circles[i, 0] -= dx * move_amount * 0.5
                            circles[i, 1] -= dy * move_amount * 0.5
                            circles[j, 0] += dx * move_amount * 0.5
                            circles[j, 1] += dy * move_amount * 0.5
                            
                            updated = True
        
        if not updated:
            break
    
    return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Voronoi-based initialization with geometric properties
    initial_points = create_voronoi_initialization(N_CIRCLES)
    
    # Compute initial radii based on Voronoi properties
    initial_radii = compute_voronoi_based_radii(initial_points, N_CIRCLES)
    
    # Create initial circles array
    circles = np.column_stack([initial_points, initial_radii])
    
    # Phase 2: Enhanced physical simulation for better distribution
    for iteration in range(300):
        # Apply boundary constraints
        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            circles[i] = [x, y, r]
        
        # Resolve overlaps with simplified repulsion
        circles = constrained_position_update(circles, max_iter=3)
        
        # Small random perturbation to escape local optima
        if iteration % 20 == 0:
            for i in range(N_CIRCLES):
                circles[i, 0] += np.random.uniform(-0.001, 0.001)
                circles[i, 1] += np.random.uniform(-0.001, 0.001)
                # Clip to bounds after perturbation
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1-circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1-circles[i, 2])
    
    # Phase 3: Optimize radii locally
    circles = local_radius_optimization(circles)
    
    # Phase 4: Evolutionary refinement with proper constraints
    
    # Custom representation: encode circle positions and radii as a flat array
    def create_individual():
        # Create initial individual from current solution
        individual = []
        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            individual.extend([x, y, r])
        return individual
    
    def evaluate_individual(individual):
        # Convert flat array back to circles
        circles_array = np.array(individual).reshape(-1, 3)
        
        # Validate and penalize invalid solutions
        if not validate_solution_jit(circles_array):
            return (-np.sum(circles_array[:, 2]) - 1000,)
        
        return (-np.sum(circles_array[:, 2]),)
    
    def mutate_individual(individual, indpb=0.2):
        """Mutate with constraint preservation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < indpb:
                if i % 3 == 2:  # radius
                    mutated[i] = max(0.001, min(0.5, mutated[i] + np.random.normal(0, 0.005)))
                else:  # position
                    mutated[i] = max(0.01, min(0.99, mutated[i] + np.random.normal(0, 0.005)))
        return mutated,
    
    def crossover_individual(ind1, ind2):
        """Simple uniform crossover"""
        size = len(ind1)
        for i in range(size):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    # Set up DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individual)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create population with better starting points
    pop = []
    for _ in range(50):
        individual = create_individual()
        # Add noise to create diversity
        for i in range(len(individual)):
            if i % 3 == 2:  # radius
                individual[i] = max(0.001, min(0.5, individual[i] + np.random.normal(0, 0.003)))
            else:  # position
                individual[i] = max(0.01, min(0.99, individual[i] + np.random.normal(0, 0.003)))
        pop.append(individual)
    
    # Evolutionary algorithm with fewer generations due to time constraints
    n_generations = 100
    for gen in range(n_generations):
        # Select parents
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = evaluate_individual(ind)
        
        # Replace population
        pop[:] = offspring
    
    # Get best solution from evolution
    best_individual = tools.selBest(pop, 1)[0]
    best_circles = np.array(best_individual).reshape(-1, 3)
    
    # Final local optimization
    best_circles = local_radius_optimization(best_circles)
    best_circles = constrained_position_update(best_circles, max_iter=5)
    
    # Final validation
    if not validate_solution_jit(best_circles):
        # If invalid, revert to physics-based solution
        return circles
    
    return best_circles

# EVOLVE-BLOCK-END