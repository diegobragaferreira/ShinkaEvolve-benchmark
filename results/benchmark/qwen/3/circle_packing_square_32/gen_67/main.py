# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial import KDTree
import math
from joblib import Parallel, delayed

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if circles are within bounds and non-overlapping."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = KDTree(points)
    
    # For each circle, check if it overlaps with others
    for i in range(n):
        x, y, r = circles[i]
        # Find nearby points (within 2*r distance)
        nearby_indices = tree.query_ball_point([x, y], 2 * r)
        
        # Check each nearby circle for overlap
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < r + r2:
                    return False
    
    return True

def get_fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii for valid configurations."""
    if not is_valid_configuration(circles):
        return 0.0
    
    return np.sum(circles[:, 2])

def initialize_population_greedy(pop_size: int, n_circles: int) -> list:
    """Initialize population using greedy heuristic approach."""
    population = []
    for _ in range(pop_size):
        circles = []
        # Try placing circles greedily
        for i in range(n_circles):
            best_circle = None
            best_score = float('inf')
            
            # Try several random placements to find good one
            for _ in range(100):
                x = random.uniform(0, 1)
                y = random.uniform(0, 1)
                r = min(x, 1-x, y, 1-y)  # Max possible radius
                
                # If too small, skip
                if r < 0.001:
                    continue
                    
                # Check if it overlaps
                valid = True
                for cx, cy, cr in circles:
                    dist_sq = (x - cx)**2 + (y - cy)**2
                    if dist_sq < (r + cr)**2:
                        valid = False
                        break
                
                if valid:
                    # Score based on how much space it takes up
                    score = -r  # Try to maximize radius
                    if score < best_score:
                        best_score = score
                        best_circle = (x, y, r)
            
            if best_circle:
                circles.append(best_circle)
            else:
                # Fallback to random placement
                x = random.uniform(0, 1)
                y = random.uniform(0, 1)
                r = min(x, 1-x, y, 1-y)
                circles.append((x, y, r))
        
        population.append(np.array(circles))
    
    return population

def local_optimize(circles: np.ndarray) -> np.ndarray:
    """Apply local optimization using simulated annealing."""
    # Simple local search around current solution
    current_circles = circles.copy()
    current_total_radius = np.sum(current_circles[:, 2])
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 1e-4
    alpha = 0.95
    max_iter = 5000
    
    for i in range(max_iter):
        # Perturb one circle
        idx = np.random.randint(len(current_circles))
        old_x, old_y, old_r = current_circles[idx]
        
        # Slightly change position and radius
        new_x = max(old_x + np.random.normal(0, 0.01), 0)
        new_y = max(old_y + np.random.normal(0, 0.01), 0)
        new_r = max(old_r + np.random.normal(0, 0.005), 0.001)
        
        # Ensure it stays within bounds
        new_x = min(new_x, 1 - new_r)
        new_y = min(new_y, 1 - new_r)
        
        # Update the circle temporarily
        temp_circles = current_circles.copy()
        temp_circles[idx] = [new_x, new_y, new_r]
        
        # Check constraints and compute new total radius
        valid = True
        penalty = 0
        
        # Check containment
        if new_x < new_r or new_x > 1 - new_r or new_y < new_r or new_y > 1 - new_r:
            penalty += 1000000
            valid = False
            
        # Check overlaps
        temp_points = temp_circles[:, :2]
        temp_radii = temp_circles[:, 2]
        
        # Check overlaps with all other circles
        for j in range(len(temp_circles)):
            if j != idx:
                x1, y1 = temp_circles[j][:2]
                r1 = temp_circles[j][2]
                x2, y2 = temp_circles[idx][:2]
                r2 = temp_circles[idx][2]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < (r1 + r2):
                    penalty += 1000000
                    valid = False
                    break
        
        if valid:
            # Accept with some probability based on temperature
            new_total_radius = np.sum(temp_circles[:, 2]) - penalty/1000000
            
            if new_total_radius > current_total_radius or \
               (T > 1e-10 and np.random.rand() < math.exp((new_total_radius - current_total_radius) / T)):
                current_circles = temp_circles
                current_total_radius = new_total_radius
        
        # Cool down
        T *= alpha
    
    return current_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        # Set seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Problem parameters
        n_circles = 32
        square_size = 1.0
        
        # Fitness function for DEAP
        def evaluate(individual):
            # Convert individual to circles array (x, y, r)
            circles = np.array(individual).reshape(-1, 3)
            
            # Check constraints
            total_radius = np.sum(circles[:, 2])
            
            # Penalty for invalid positions
            penalty = 0
            
            # Check containment constraints
            for i in range(n_circles):
                x, y, r = circles[i]
                if x < r or x > 1 - r or y < r or y > 1 - r:
                    penalty += 1000000
            
            # Check overlap constraints using KDTree for efficiency
            points = circles[:, :2]
            radii = circles[:, 2]
            
            # Build KDTree once for all checks
            tree = KDTree(points)
            
            # Check pairwise overlaps
            for i in range(n_circles):
                x1, y1, r1 = circles[i]
                # Find neighbors within distance 2*r1 (potential overlaps)
                neighbors = tree.query_ball_point([x1, y1], 2 * r1)
                for j in neighbors:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if distance < (r1 + r2):  # Overlap detected
                            penalty += 1000000
            
            # Return fitness (negative because we want to maximize)
            return (-total_radius - penalty,),
        
        # Create classes for GA
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_circles * 3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Initialize population with greedy heuristic
        pop = initialize_population_greedy(50, n_circles)
        hof = tools.HallOfFame(1)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Evolution parameters
        CXPB = 0.8
        MUTPB = 0.1
        NGEN = 100
        
        # Run evolution with early stopping
        try:
            pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, 
                                              stats=stats, halloffame=hof, verbose=False)
        except Exception as e:
            # Fallback to basic solution if evolution fails
            print(f"Evolution failed with error: {e}")
            # Return simple configuration
            result = np.zeros((n_circles, 3))
            # Place circles along diagonal
            for i in range(n_circles):
                r = 0.02  # Small radius for initial placement
                x = 0.05 + i * 0.03
                y = 0.05 + i * 0.03
                result[i] = [x, y, r]
            return result
        
        # Extract best solution
        best_individual = hof[0]
        circles_array = np.array(best_individual).reshape(-1, 3)
        
        # Apply local optimization using simulated annealing
        optimized_circles = local_optimize(circles_array)
        
        return optimized_circles
        
    except Exception as e:
        # On error, fallback to basic configuration
        print(f"Error during circle packing: {e}")
        circles = np.zeros((32, 3))
        for i in range(32):
            circles[i] = [0.1 + i * 0.03, 0.1 + (i % 4) * 0.1, 0.05]
        return circles

# EVOLVE-BLOCK-END
